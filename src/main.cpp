#include <iostream>
#include <vector>
#include "raylib.h"
#include "raymath.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "kan/kan.h"

const float lineThickness = 2.0f;  // Stroke size
const float squareSize = 60.0f;    // Square size
const float canvasRatio = 0.7f;
const uint32_t splineNumPoints = 20;

KAN::KANNet net;

enum KANGUIState { MENU, EDIT, TRAIN };

struct Node {
  Vector2 position;
};

struct EditState {
  Vector2 scroll = {0, 0};
  int spline_order;
  int grid_size;
  std::vector<int32_t> widths;
};

std::vector<std::vector<std::vector<std::vector<float>>>> splinesData;
std::vector<std::vector<float>> min_act;
std::vector<std::vector<float>> max_act;

EditState editState;

bool IsMouseOverRectangle(Vector2 mousePosition,
                          Vector2 nodePosition,
                          float squareSize) {
  Rectangle rect = {nodePosition.x - squareSize / 2,
                    nodePosition.y - squareSize / 2, squareSize, squareSize};
  return CheckCollisionPointRec(mousePosition, rect);
}

void CalcSplineData(KAN::KANNet& net);

void InitKANNet() {
  editState.widths.resize(net.num_layers + 1);
  for (int i = 0; i < net.num_layers; i++) {
    editState.widths[i] = net.layers[i].in_features;
  }
  editState.widths[net.num_layers] =
      net.layers[net.num_layers - 1].out_features;
  editState.spline_order = net.spline_order;
  editState.grid_size = net.grid.shape[0] - 2 * net.spline_order - 1;

  splinesData.resize(net.num_layers);
  min_act.resize(net.num_layers + 1);
  max_act.resize(net.num_layers + 1);
  min_act[0].assign(net.layers[0].in_features, -1);
  max_act[0].assign(net.layers[0].in_features, 1);
  for (uint32_t l = 0; l < net.num_layers; l++) {
    min_act[l + 1].assign(net.layers[l].out_features, -1);
    max_act[l + 1].assign(net.layers[l].out_features, 1);
    splinesData[l].resize(net.layers[l].out_features);
    for (uint32_t j = 0; j < net.layers[l].out_features; j++) {
      splinesData[l][j].resize(net.layers[l].in_features);
      for (uint32_t i = 0; i < net.layers[l].in_features; i++) {
        splinesData[l][j][i].resize(splineNumPoints);
      }
    }
  }

  CalcSplineData(net);
}

std::vector<std::vector<Node>> InitKANCanvas(std::vector<int>& widths,
                                             int x,
                                             int y,
                                             float canvasWidth,
                                             float canvasHeight) {
  // Define the network structure
  const int numLayers = widths.size() - 1;
  const int realNumLayers = 2 * numLayers + 1;
  std::vector<std::vector<Node>> layers(realNumLayers);

  int maxWidth = -1;
  for (int i = 0; i < numLayers; i++)
    maxWidth = std::max(maxWidth, widths[i] * widths[i + 1]);

  canvasWidth = std::max(canvasWidth, 1.2f * maxWidth * squareSize);
  canvasHeight = std::max(canvasHeight, 1.5f * squareSize * realNumLayers);

  for (int i = 0; i < numLayers; i++) {
    // Circle layer
    int numNodes = widths[i];
    float layerHeight =
        canvasHeight - canvasHeight / realNumLayers * (2 * i + 0.5f);
    float spacing = canvasWidth / numNodes;

    for (int j = 0; j < numNodes; j++) {
      Node node;
      node.position = {spacing * (j + 0.5f), layerHeight};
      layers[2 * i].push_back(node);
    }

    // Square layer
    numNodes = widths[i] * widths[i + 1];
    layerHeight =
        canvasHeight - canvasHeight / realNumLayers * (2 * i + 1 + 0.5f);
    spacing = canvasWidth / numNodes;
    for (int j = 0; j < numNodes; j++) {
      Node node;
      node.position = {spacing * (j + 0.5f), layerHeight};
      layers[2 * i + 1].push_back(node);
    }

    if (i == numLayers - 1) {
      // Circle layer
      numNodes = widths[i + 1];
      layerHeight =
          canvasHeight - canvasHeight / realNumLayers * (2 * i + 2 + 0.5f);
      spacing = canvasWidth / numNodes;
      for (int j = 0; j < numNodes; j++) {
        Node node;
        node.position = {spacing * (j + 0.5f), layerHeight};
        layers[2 * i + 2].push_back(node);
      }
    }
  }

  return layers;
}

std::vector<std::vector<Node>> InitKANCanvas(KAN::KANNet& net,
                                             int x,
                                             int y,
                                             float canvasWidth,
                                             float canvasHeight) {
  std::vector<int> widths(net.num_layers + 1);
  for (int i = 0; i < net.num_layers; i++) {
    widths[i] = net.layers[i].in_features;
  }
  widths[net.num_layers] = net.layers[net.num_layers - 1].out_features;
  return InitKANCanvas(widths, x, y, canvasWidth, canvasHeight);
}

void DrawKANNet(std::vector<std::vector<Node>>& layers,
                Camera2D& camera,
                Vector2 mousePosition) {
  // Draw connections
  for (int i = 0; i < layers.size() / 2; i++) {
    int curWidth = layers[2 * i].size();
    int nextWidth = layers[2 * i + 2].size();
    for (int j = 0; j < curWidth; j++) {
      for (int k = 0; k < nextWidth; k++) {
        DrawLineEx(layers[2 * i][j].position,
                   Vector2Add(layers[2 * i + 1][j * nextWidth + k].position,
                              Vector2{0, squareSize / 2}),
                   lineThickness * 1.5, BLACK);
        DrawLineEx(
            Vector2Subtract(layers[2 * i + 1][j * nextWidth + k].position,
                            Vector2{0, squareSize / 2}),
            layers[2 * i + 2][k].position, lineThickness * 1.5, BLACK);
      }
    }
  }

  // Draw nodes
  for (int i = 0; i < layers.size(); i++) {
    for (int j = 0; j < layers[i].size(); j++) {
      Node node = layers[i][j];
      // Draw the nodes (circles for even layers, squares for odd layers)
      if (i % 2 == 0) {
        DrawCircleV(node.position, 15, BLACK);  // Increased circle size
      } else {
        // Draw the square with a white fill and black border
        DrawRectangle(node.position.x - squareSize / 2,
                      node.position.y - squareSize / 2, squareSize, squareSize,
                      WHITE);
        DrawRectangleLinesEx((Rectangle){node.position.x - squareSize / 2,
                                         node.position.y - squareSize / 2,
                                         squareSize, squareSize},
                             lineThickness, BLACK);
        // Check for click on this square
        Vector2 worldMousePosition =
            Vector2Subtract(mousePosition, camera.offset);
        worldMousePosition =
            Vector2Scale(worldMousePosition, 1.0f / camera.zoom);
        worldMousePosition = Vector2Add(worldMousePosition, camera.target);

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
            IsMouseOverRectangle(worldMousePosition, node.position,
                                 squareSize)) {
          std::cout << "Square node clicked at: (" << node.position.x << ", "
                    << node.position.y << ")\n";
        }
      }
    }
  }
}

void DrawSineCurve(float x, float y, float w, float h) {
  // Number of points to draw the sine wave (more points = smoother curve)
  const int numPoints = 1000;
  x += 4;
  y += 4;
  w -= 8;
  h -= 8;

  // Step between each point (in radians, for one full sine wave cycle)
  float step = (2.0f * 3.14) / (float)numPoints;

  // Loop to draw the sine wave
  for (int i = 0; i < numPoints; i++) {
    // Calculate current and next point positions
    float currentX = x + (i / (float)numPoints) * w;
    float nextX = x + ((i + 1) / (float)numPoints) * w;

    // Map sine wave from [-1, 1] to [y, y+h]
    float currentY = y + (h / 2) * (1 - sin(i * step));  // Midpoint is y + h/2
    float nextY =
        y +
        (h / 2) * (1 - sin((i + 1) *
                           step));  // Invert sine output to fit in the square

    // Draw line between the two points
    DrawLineEx(Vector2{currentX, currentY}, Vector2{nextX, nextY}, 3, RED);
  }
}

RenderTexture DrawSineCurveToTexture(int textureWidth, int textureHeight) {
  // Create a render texture
  RenderTexture texture = LoadRenderTexture(textureWidth, textureHeight);

  // Start drawing to the texture
  BeginTextureMode(texture);
  ClearBackground(BLANK);  // Make background transparent

  const int numPoints = 100;
  float step = (6.283f) / (float)numPoints;

  for (int i = 0; i < numPoints; i++) {
    float currentX = (i / (float)numPoints) * textureWidth;
    float nextX = ((i + 1) / (float)numPoints) * textureWidth;

    float currentY = (textureHeight / 2) * (1 - sin(i * step));
    float nextY = (textureHeight / 2) * (1 - sin((i + 1) * step));

    DrawLineEx(Vector2{currentX, currentY}, Vector2{nextX, nextY}, 3, RED);
  }

  EndTextureMode();  // Stop drawing to the texture

  return texture;
}

void DrawSpline(Rectangle bound, uint32_t l, uint32_t i, uint32_t j) {
  float x = bound.x + 4;
  float y = bound.y + 4;
  float w = bound.width - 8;
  float h = bound.height - 8;

  float minX = min_act[l][j];
  float maxX = max_act[l][j];
  float minY = splinesData[l][j][i][0];
  float maxY = splinesData[l][j][i][0];

  for (uint32_t k = 0; k < splineNumPoints; k++) {
    minY = std::min(minY, splinesData[l][j][i][k]);
    maxY = std::max(maxY, splinesData[l][j][i][k]);
  }

  // Loop to draw the spline
  for (uint32_t k = 0; k < splineNumPoints - 1; k++) {
    // Calculate current and next point positions
    float currentX = x + (w / ((float)splineNumPoints - 1)) * k;
    float nextX = x + (w / ((float)splineNumPoints - 1)) * (k + 1);

    float currentY =
        y + h - (splinesData[l][j][i][k] - minY) / (maxY - minY) * h;
    float nextY =
        y + h - (splinesData[l][j][i][k + 1] - minY) / (maxY - minY) * h;

    // Draw line between the two points
    DrawLineEx(Vector2{currentX, currentY}, Vector2{nextX, nextY}, 3, RED);
  }
}

void CalcSplineData(KAN::KANNet& net) {
  for (uint32_t l = 0; l < net.num_layers; l++) {
    KAN::Tensor X({splineNumPoints, net.layers[l].in_features});
    for (uint32_t i = 0; i < net.layers[l].in_features; i++)
      X(0, i) = min_act[l][i];

    for (uint32_t i = 1; i < splineNumPoints; ++i)
      for (uint32_t j = 0; j < net.layers[l].in_features; j++)
        X(i, j) = X(i - 1, j) +
                  (max_act[l][j] - min_act[l][j]) / (splineNumPoints - 1);

    // X.view(-1)
    X.dim = 1;
    X.shape[0] = X.shape[0] * X.shape[1];
    X.stride[0] = 1;

    KAN::Tensor bases =
        KAN::Tensor({X.shape[0], net.grid.shape[0] - net.spline_order - 1});
    KAN::Tensor bases_temp = KAN::Tensor({X.shape[0], net.grid.shape[0] - 1});
    KAN::b_splines(net.grid, X, net.spline_order, &bases, &bases_temp, nullptr);

    for (uint32_t i = 0; i < net.layers[l].in_features; i++) {
      for (uint32_t j = 0; j < net.layers[l].out_features; j++) {
        for (uint32_t k = 0; k < splineNumPoints; k++) {
          float accumulator = 0;
          for (uint32_t t = 0; t < bases.shape[1]; t++)
            accumulator += net.layers[l].coeff(j, i, t) *
                           bases(k * net.layers[l].in_features + i, t);
          splinesData[l][j][i][k] =
              net.layers[l].spline_weights(j, i) * accumulator +
              net.layers[l].basis_weights(j, i) *
                  KAN::SiLU(X(k * net.layers[l].in_features + i));
        }
      }
    }
  }
}

void DrawKANSplines(std::vector<std::vector<Node>>& layers, KAN::KANNet& net) {
  for (uint32_t l = 0; l < net.num_layers; l++) {
    for (uint32_t i = 0; i < net.layers[l].in_features; i++) {
      for (uint32_t j = 0; j < net.layers[l].out_features; j++) {
        Vector2 position =
            layers[2 * l + 1][i * net.layers[l].out_features + j].position;
        position =
            Vector2Subtract(position, Vector2{squareSize / 2, squareSize / 2});
        // RenderTexture sineTexture =
        //     DrawSineCurveToTexture(squareSize, squareSize);
        // DrawTextureRec(
        //     sineTexture.texture, Rectangle{0, 0, squareSize, squareSize},
        //     (Vector2){position.x - squareSize / 2, position.y - squareSize /
        //     2}, WHITE);

        // UnloadRenderTexture(sineTexture);
        // DrawSineCurve(position.x - squareSize / 2, position.y - squareSize /
        // 2,
        //               squareSize, squareSize);
        // DrawRectangleLinesEx(
        //     (Rectangle){position.x - squareSize / 2,
        //                 position.y - squareSize / 2, squareSize, squareSize},
        //     lineThickness, BLACK);

        DrawSpline((Rectangle){position.x, position.y, squareSize, squareSize},
                   l, i, j);
      }
    }
  }
}

void DrawMenuGUI(float panelWidth,
                 float panelX,
                 float panelY,
                 KANGUIState& kanGuiState) {
  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Edit Architecture")) {
    kanGuiState = EDIT;
    editState.spline_order = net.spline_order;
    editState.grid_size = net.grid.shape[0] - (2 * net.spline_order + 1);
    editState.widths.resize(net.num_layers + 1);
    for (int i = 0; i < net.num_layers; i++) {
      editState.widths[i] = net.layers[i].in_features;
    }
    editState.widths[net.num_layers] =
        net.layers[net.num_layers - 1].out_features;
  }
  panelY += 50;
  GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
            "Load Checkpoint");
  panelY += 50;
  GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
            "Train Network");
  panelY += 50;
}

void DrawEditGUI(float panelWidth,
                 float panelX,
                 float panelY,
                 KANGUIState& kanGuiState,
                 KAN::KANNet& net) {
  GuiLabel(Rectangle{panelX + 10, panelY, 125, 30}, "Spline Order");
  GuiSpinner((Rectangle){panelX + 135, panelY, 125, 30}, NULL,
             &editState.spline_order, 2, 10, false);
  panelY += 40;
  GuiLabel(Rectangle{panelX + 10, panelY, 125, 30}, "Grid Size");
  GuiSpinner((Rectangle){panelX + 135, panelY, 125, 30}, NULL,
             &editState.grid_size, 1, 10, false);
  panelY += 40;

  // Add Layer button
  Color originalColor = GetColor(GuiGetStyle(BUTTON, BASE_COLOR_NORMAL));
  Color originalTextColor = GetColor(GuiGetStyle(BUTTON, TEXT_COLOR_NORMAL));
  GuiSetStyle(BUTTON, BASE_COLOR_NORMAL, ColorToInt(GREEN));
  GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 30},
                "Add Layer")) {
    editState.widths.push_back(1);  // Add a new layer with a default value of 1
  }
  GuiSetStyle(BUTTON, BASE_COLOR_NORMAL, ColorToInt(originalColor));
  GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL, ColorToInt(originalTextColor));
  panelY += 40;

  // Scroll Panel
  Rectangle scrollPanel = {panelX + 10, panelY, panelWidth - 20, 400};
  Rectangle content = {0, 0, panelWidth - 40,
                       editState.widths.size() * 40.0f + 10};
  Rectangle view = {0};

  GuiScrollPanel(scrollPanel, NULL, content, &editState.scroll, &view);
  BeginScissorMode(scrollPanel.x, scrollPanel.y, scrollPanel.width,
                   scrollPanel.height);
  for (int i = 0; i < editState.widths.size(); i++) {
    Rectangle textBoxRect = {
        scrollPanel.x + 10,
        scrollPanel.y + 10 + i * (30 + 10) + editState.scroll.y,
        panelWidth - 90, 30};  // Adjusted width for spinner

    GuiSpinner(textBoxRect, NULL,
               &editState.widths[editState.widths.size() - i - 1], 1, 10,
               false);

    // Minus button to delete the layer
    Rectangle minusButtonRect = {textBoxRect.x + textBoxRect.width + 10,
                                 textBoxRect.y, 30, 30};
    if (editState.widths.size() < 3)
      GuiDisable();

    Color originalColor = GetColor(GuiGetStyle(BUTTON, BASE_COLOR_NORMAL));
    Color originalTextColor = GetColor(GuiGetStyle(BUTTON, TEXT_COLOR_NORMAL));
    GuiSetStyle(BUTTON, BASE_COLOR_NORMAL, ColorToInt(RED));
    GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
    if (GuiButton(minusButtonRect, "-")) {
      editState.widths.erase(editState.widths.begin() +
                             (editState.widths.size() - i - 1));
    }
    GuiSetStyle(BUTTON, BASE_COLOR_NORMAL, ColorToInt(originalColor));
    GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL, ColorToInt(originalTextColor));
    GuiEnable();
  }
  EndScissorMode();
  panelY += 410;

  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Save Changes")) {
    std::vector<uint32_t> convertedWidths(editState.widths.begin(),
                                          editState.widths.end());
    net = KAN::KANNet_create(convertedWidths, editState.spline_order,
                             editState.grid_size);
    InitKANNet();
    kanGuiState = MENU;
  }
  panelY += 50;

  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Discard Changes"))
    kanGuiState = MENU;
  panelY += 50;
}

int main() {
  uint32_t num_layers, spline_order, grid_size, *widths;
  float* params_data;
  KAN::KANNet_load_checkpoint("checkpoint2.dat", num_layers, spline_order,
                              grid_size, widths, params_data);
  net = KAN::KANNet_create(std::vector(widths, widths + num_layers + 1),
                           spline_order, grid_size, params_data);
  InitKANNet();

  // Initialization
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(1200, 800, "Dynamic Network Example");
  SetTargetFPS(60);
  GuiSetStyle(DEFAULT, TEXT_SIZE, 20);

  float screenWidth = GetScreenWidth();
  float screenHeight = GetScreenHeight();
  float canvasWidth = screenWidth * canvasRatio;
  auto layers = InitKANCanvas(net, 0, 0, canvasWidth, screenHeight);

  // Initialize camera
  Camera2D camera = {0};
  camera.target = (Vector2){canvasWidth / 2.0f, screenHeight / 2.0f};
  camera.offset = (Vector2){canvasWidth / 2.0f, screenHeight / 2.0f};
  camera.rotation = 0.0f;
  camera.zoom = 1.0f;

  Vector2 lastMousePosition = GetMousePosition();
  bool isDragging = false;
  bool inBound = false;

  KANGUIState kanGuiState = MENU;

  while (!WindowShouldClose())  // Main game loop
  {
    camera.target.x /= canvasWidth;
    camera.target.y /= screenHeight;

    screenWidth = GetScreenWidth();
    screenHeight = GetScreenHeight();
    canvasWidth = screenWidth * canvasRatio;

    camera.target.x *= canvasWidth;
    camera.target.y *= screenHeight;
    camera.offset = (Vector2){canvasWidth / 2.0f, screenHeight / 2.0f};

    if (kanGuiState == EDIT)
      layers = InitKANCanvas(editState.widths, 0, 0, canvasWidth, screenHeight);
    else
      layers = InitKANCanvas(net, 0, 0, canvasWidth, screenHeight);

    Vector2 mousePosition = GetMousePosition();
    inBound = CheckCollisionPointRec(mousePosition,
                                     {0, 0, canvasWidth, screenHeight});
    if (inBound) {
      // Handle panning
      if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        isDragging = true;
      }
      if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
        isDragging = false;
      }

      if (isDragging) {
        Vector2 delta = Vector2Subtract(lastMousePosition, mousePosition);
        camera.target =
            Vector2Add(camera.target, Vector2Scale(delta, 1.0f / camera.zoom));
      }

      lastMousePosition = mousePosition;

      // Handle zooming
      float zoomDelta = GetMouseWheelMove() * 0.1f;
      camera.zoom += zoomDelta;
      camera.zoom = std::min(2.0f, std::max(0.2f, camera.zoom));
    } else {
      isDragging = false;
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode2D(camera);
    DrawKANNet(layers, camera, mousePosition);
    if (kanGuiState != EDIT)
      DrawKANSplines(layers, net);
    EndMode2D();

    float roundness = 0.2f;
    float width = 200.0f;
    float height = 100.0f;
    float segments = 0.0f;
    float lineThick = 1.0f;
    bool drawRect = false;
    bool drawRoundedRect = true;
    bool drawRoundedLines = false;

    float panelWidth = screenWidth - canvasWidth;
    DrawLine(canvasWidth, 0, canvasWidth, screenHeight, Fade(LIGHTGRAY, 0.6f));
    DrawRectangle(canvasWidth, 0, panelWidth, screenHeight,
                  (Color){235, 235, 235, 255});

    float panelX = canvasWidth;
    float panelY = 10;

    // Draw GUI controls
    //------------------------------------------------------------------------------
    Vector2 textDim = MeasureTextEx(GetFontDefault(), "KAN Visualizer", 40, 2);
    DrawTextEx(GetFontDefault(), "KAN Visualizer",
               (Vector2){panelX + (panelWidth - textDim.x) / 2, panelY}, 40, 2,
               BLUE);
    panelY += textDim.y + 20;

    switch (kanGuiState) {
      case MENU:
        DrawMenuGUI(panelWidth, panelX, panelY, kanGuiState);
        break;
      case EDIT:
        DrawEditGUI(panelWidth, panelX, panelY, kanGuiState, net);
      default:
        break;
    }
    //------------------------------------------------------------------------------

    DrawFPS(10, 10);

    EndDrawing();
  }

  // De-Initialization
  CloseWindow();

  return 0;
}

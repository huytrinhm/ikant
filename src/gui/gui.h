#ifndef GUI_H
#define GUI_H

#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include "raylib.h"
#include "raymath.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "tinyfiledialogs.h"

#include "../kan/kan.h"
#include "../kan/spline.h"

const float lineThickness = 2.0f;  // Stroke size
const float squareSize = 60.0f;    // Square size
const float canvasRatio = 0.7f;
const uint32_t splineNumPoints = 20;

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

struct PopupState {
  bool opened;
  uint32_t l;
  uint32_t i;
  uint32_t j;
};

EditState editState;
PopupState popupState;

bool IsMouseOverRectangle(Vector2 mousePosition,
                          Vector2 nodePosition,
                          float squareSize) {
  Rectangle rect = {nodePosition.x - squareSize / 2,
                    nodePosition.y - squareSize / 2, squareSize, squareSize};
  return CheckCollisionPointRec(mousePosition, rect);
}

void CalcSplineData(
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    bool updateAlpha);

void InitKANNet(
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::atomic<int>& currentEpoch,
    float& loss) {
  editState.widths.resize(net.num_layers + 1);
  for (int i = 0; i < net.num_layers; i++) {
    editState.widths[i] = net.layers[i].in_features;
  }
  editState.widths[net.num_layers] =
      net.layers[net.num_layers - 1].out_features;
  editState.spline_order = net.spline_order;
  editState.grid_size = net.grid.shape[0] - 2 * net.spline_order - 1;

  splinesData.resize(net.num_layers);
  splinesAlpha.resize(net.num_layers);
  min_act.resize(net.num_layers + 1);
  max_act.resize(net.num_layers + 1);
  min_act[0].assign(net.layers[0].in_features, -1);
  max_act[0].assign(net.layers[0].in_features, 1);
  for (uint32_t l = 0; l < net.num_layers; l++) {
    min_act[l + 1].assign(net.layers[l].out_features, -1);
    max_act[l + 1].assign(net.layers[l].out_features, 1);
    splinesData[l].resize(net.layers[l].out_features);
    splinesAlpha[l].resize(net.layers[l].out_features);
    for (uint32_t j = 0; j < net.layers[l].out_features; j++) {
      splinesData[l][j].resize(net.layers[l].in_features);
      splinesAlpha[l][j].assign(net.layers[l].in_features, 1);
      for (uint32_t i = 0; i < net.layers[l].in_features; i++) {
        splinesData[l][j][i].resize(splineNumPoints);
      }
    }
  }

  currentEpoch = 0;
  loss = -1;

  popupState.opened = false;

  CalcSplineData(net, splinesData, splinesAlpha, min_act, max_act, false);
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

void DrawKANNet(KAN::KANNet& net,
                std::vector<std::vector<Node>>& layers,
                std::vector<std::vector<std::vector<float>>>& splinesAlpha,
                Camera2D& camera,
                Vector2 mousePosition,
                bool isEditing) {
  // Draw connections
  for (int i = 0; i < layers.size() / 2; i++) {
    int curWidth = layers[2 * i].size();
    int nextWidth = layers[2 * i + 2].size();
    for (int j = 0; j < curWidth; j++) {
      for (int k = 0; k < nextWidth; k++) {
        // std::cout << splinesAlpha[i][k][j] << std::endl;
        unsigned char alpha =
            isEditing ? 255 : (unsigned char)(255 * splinesAlpha[i][k][j]);
        DrawLineEx(layers[2 * i][j].position,
                   Vector2Add(layers[2 * i + 1][j * nextWidth + k].position,
                              Vector2{0, squareSize / 2}),
                   lineThickness * 1.5,
                   Color{
                       0,
                       0,
                       0,
                       alpha,
                   });
        DrawLineEx(
            Vector2Subtract(layers[2 * i + 1][j * nextWidth + k].position,
                            Vector2{0, squareSize / 2}),
            layers[2 * i + 2][k].position, lineThickness * 1.5,
            Color{
                0,
                0,
                0,
                alpha,
            });
      }
    }
  }

  // Draw nodes
  for (int i = 0; i < layers.size(); i++) {
    for (int j = 0; j < layers[i].size(); j++) {
      Node node = layers[i][j];
      Vector2 worldMousePosition =
          Vector2Subtract(mousePosition, camera.offset);
      worldMousePosition = Vector2Scale(worldMousePosition, 1.0f / camera.zoom);
      worldMousePosition = Vector2Add(worldMousePosition, camera.target);
      // Draw the nodes (circles for even layers, squares for odd layers)
      if (i % 2 == 0) {
        Color color =
            (i == 0 || isEditing)
                ? BLACK
                : Color{0, 0, 0,
                        (unsigned char)(net.layers[i / 2 - 1].mask(j) * 127 +
                                        128)};
        DrawCircleV(node.position, 15, color);  // Increased circle size
        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON) &&
            IsMouseOverRectangle(worldMousePosition, node.position, 30) &&
            i != 0 && i != layers.size() - 1 && !isEditing &&
            !popupState.opened) {
          std::cout << "Circle node clicked at: (" << i / 2 - 1 << ", " << j
                    << ")\n";
          net.layers[i / 2 - 1].mask(j) = 1 - net.layers[i / 2 - 1].mask(j);
        }
      } else {
        // Draw the square with a white fill and black border
        unsigned char alpha =
            isEditing
                ? 255
                : (unsigned char)(255 *
                                  splinesAlpha[i / 2][j % layers[i + 1].size()]
                                              [j / layers[i + 1].size()]);
        DrawRectangle(node.position.x - squareSize / 2,
                      node.position.y - squareSize / 2, squareSize, squareSize,
                      Color{
                          255,
                          255,
                          255,
                          alpha,
                      });
        DrawRectangleLinesEx((Rectangle){node.position.x - squareSize / 2,
                                         node.position.y - squareSize / 2,
                                         squareSize, squareSize},
                             lineThickness,
                             Color{
                                 0,
                                 0,
                                 0,
                                 alpha,
                             });
        // Check for click on this square
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
            IsMouseOverRectangle(worldMousePosition, node.position,
                                 squareSize) &&
            !isEditing && !popupState.opened) {
          // std::cout << "Square node clicked at: (" << node.position.x << ", "
          //           << node.position.y << ")\n";

          popupState.opened = true;
          popupState.l = i / 2;
          popupState.i = j % layers[i + 1].size();
          popupState.j = j / layers[i + 1].size();
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

void DrawSpline(
    Rectangle bound,
    uint32_t l,
    uint32_t i,
    uint32_t j,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex) {
  splinesDataMutex.lock();
  // std::cout << l << i << j << std::endl;
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
  splinesDataMutex.unlock();
}

void DrawSpline(
    Rectangle bound,
    uint32_t l,
    uint32_t i,
    uint32_t j,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex) {
  splinesDataMutex.lock();
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
    DrawLineEx(
        Vector2{currentX, currentY}, Vector2{nextX, nextY}, 3,
        Color{230, 41, 55, (unsigned char)(splinesAlpha[l][j][i] * 255)});
  }
  splinesDataMutex.unlock();
}

float scoreToAlpha(float score) {
  float alpha = std::tanh(2.5 * score);
  return alpha;
}

void CalcSplineData(
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    bool updateAlpha) {
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

        if (updateAlpha)  // Update alpha values
          splinesAlpha[l][j][i] =
              scoreToAlpha(net.layers[l].edge_activations(j, i));
      }
    }
  }
}

void DrawKANSplines(
    std::vector<std::vector<Node>>& layers,
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex) {
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
                   l, i, j, splinesData, splinesAlpha, min_act, max_act,
                   splinesDataMutex);
      }
    }
  }
}

void DrawKANSplinePopup(
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex) {
  if (!popupState.opened)
    return;

  float canvasHeight = GetScreenHeight();
  float canvasWidth = GetScreenWidth() * canvasRatio;
  float windowSize = std::min(canvasWidth, canvasHeight) * 0.8f;
  float windowX = (canvasWidth - windowSize) / 2;
  float windowY = (canvasHeight - windowSize) / 2;
  popupState.opened = !GuiWindowBox(
      Rectangle{windowX, windowY, windowSize, windowSize + 20}, "Spline");

  DrawSpline((Rectangle){windowX, windowY + 20, windowSize, windowSize},
             popupState.l, popupState.j, popupState.i, splinesData, min_act,
             max_act, splinesDataMutex);
}

void HandleLoadCheckpoint(
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::atomic<int>& currentEpoch,
    float& loss) {
  char* filename =
      tinyfd_openFileDialog("Load Checkpoint...", "", 0, NULL, NULL, 0);
  if (!filename)
    return;
  uint32_t num_layers, spline_order, grid_size, *widths;
  float* params_data;
  KAN::KANNet_load_checkpoint(filename, num_layers, spline_order, grid_size,
                              widths, params_data);
  net = KAN::KANNet_create(std::vector(widths, widths + num_layers + 1),
                           spline_order, grid_size, params_data);
  InitKANNet(net, splinesData, splinesAlpha, min_act, max_act, currentEpoch,
             loss);
}

void DrawMenuGUI(
    float panelWidth,
    float panelX,
    float panelY,
    KANGUIState& kanGuiState,
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::atomic<int>& currentEpoch,
    float& loss) {
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
  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Load Checkpoint")) {
    HandleLoadCheckpoint(net, splinesData, splinesAlpha, min_act, max_act,
                         currentEpoch, loss);
  }
  panelY += 50;
  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Train Network")) {
    kanGuiState = TRAIN;
  }
  panelY += 50;
}

void DrawEditGUI(
    float panelWidth,
    float panelX,
    float panelY,
    KANGUIState& kanGuiState,
    KAN::KANNet& net,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::atomic<int>& currentEpoch,
    float& loss) {
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
    InitKANNet(net, splinesData, splinesAlpha, min_act, max_act, currentEpoch,
               loss);
    kanGuiState = MENU;
  }
  panelY += 50;

  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Discard Changes"))
    kanGuiState = MENU;
  panelY += 50;
}

void RunOneEpoch(
    float lr,
    float lambda,
    KAN::KANNet& net,
    float& loss,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex,
    KAN::Tensor& X,
    KAN::Tensor& y) {
  KAN::KANNet_zero_grad(net);
  for (uint32_t l = 0; l < net.num_layers; l++)
    for (uint32_t i = 0; i < net.layers[l].out_features; i++)
      for (uint32_t j = 0; j < net.layers[l].in_features; j++)
        net.layers[l].edge_activations(i, j) = 0;
  float epoch_loss = 0;

  for (uint32_t i = 0; i < X.shape[0]; ++i) {
    KAN::Tensor sample(1, &X.shape[1], &X.stride[1], &X(i, 0));
    KAN::Tensor gt(1, &y.shape[1], &y.stride[1], &y(i, 0));
    KANNet_forward(net, sample);
    KANNet_backward(net, sample, gt, lambda);
    epoch_loss += MSELoss(net.layers[net.num_layers - 1].activations, gt);
    KAN::KANNet_get_spline_range(net, sample, min_act, max_act, i == 0);
  }

  for (uint32_t l = 0; l < net.num_layers; l++)
    for (uint32_t i = 0; i < net.layers[l].out_features; i++)
      for (uint32_t j = 0; j < net.layers[l].in_features; j++)
        net.layers[l].edge_activations(i, j) /= X.shape[0];

  std::lock_guard<std::mutex> guard(splinesDataMutex);
  CalcSplineData(net, splinesData, splinesAlpha, min_act, max_act, true);

  for (uint32_t i = 0; i < net.num_params; i++) {
    net.params_data[i] -= lr * net.params_grad_data[i] / X.shape[0];
  }

  loss = epoch_loss / X.shape[0];
}

void RunTraining(
    float lr,
    float lambda,
    int epoch,
    std::atomic<bool>& training,
    std::atomic<int>& currentEpoch,
    KAN::KANNet& net,
    float& loss,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex,
    KAN::Tensor& X,
    KAN::Tensor& y) {
  training = true;
  for (; currentEpoch < epoch && training; currentEpoch++) {
    RunOneEpoch(lr, lambda, net, loss, splinesData, splinesAlpha, min_act,
                max_act, splinesDataMutex, X, y);
  }
  training = false;
}

void HandleLoadData(KAN::Tensor& X, KAN::Tensor& y) {
  char* filename = tinyfd_openFileDialog("Load Data...", "", 0, NULL, NULL, 0);
  if (!filename)
    return;
  std::ifstream file(filename, std::ios::binary);
  X = KAN::tensor_from_filestream(file);
  y = KAN::tensor_from_filestream(file);
  file.close();
}

void HandleSaveCheckpoint(KAN::KANNet& net) {
  char* filename =
      tinyfd_saveFileDialog("Save Checkpoint...", "", 0, NULL, NULL);
  if (!filename)
    return;
  KAN::KANNet_save_checkpoint(net, filename);
}

void DrawTrainGUI(
    float panelWidth,
    float panelX,
    float panelY,
    KANGUIState& kanGuiState,
    KAN::KANNet& net,
    KAN::Tensor& X,
    KAN::Tensor& y,
    std::atomic<bool>& training,
    float& loss,
    std::atomic<int>& currentEpoch,
    std::thread& trainThread,
    std::vector<std::vector<std::vector<std::vector<float>>>>& splinesData,
    std::vector<std::vector<std::vector<float>>>& splinesAlpha,
    std::vector<std::vector<float>>& min_act,
    std::vector<std::vector<float>>& max_act,
    std::mutex& splinesDataMutex) {
  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Load Data")) {
    HandleLoadData(X, y);
  }
  panelY += 50;

  std::string dataStr = "Data: ";
  if (X.dim)
    dataStr += std::to_string(X.shape[0]) + " samples, " +
               std::to_string(X.shape[1]) + " features";
  else
    dataStr += "<Not loaded>";
  GuiLabel(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
           dataStr.c_str());
  panelY += 50;

  // Variables to hold the input values
  static float lr = 0.01f;
  static bool lrEditMode = false;
  static char lrTextValue[32] = {'0', '.', '0', '1', 0};

  static float lambda = 0.0f;
  static bool lambdaEditMode = false;
  static char lambdaTextValue[32] = {'0', 0};

  static int epoch = 100;
  static bool epochEditMode = false;

  // Input fields for lr, lambda, and epoch
  GuiLabel(Rectangle{panelX + 10, panelY, 160, 40}, "Learning Rate:");
  if (GuiValueBoxFloat(Rectangle{panelX + 180, panelY, 100, 40}, nullptr,
                       lrTextValue, &lr, lrEditMode)) {
    lrEditMode = !lrEditMode;
  }
  panelY += 50;

  GuiLabel(Rectangle{panelX + 10, panelY, 160, 40}, "Lambda:");
  if (GuiValueBoxFloat(Rectangle{panelX + 180, panelY, 100, 40}, nullptr,
                       lambdaTextValue, &lambda, lambdaEditMode)) {
    lambdaEditMode = !lambdaEditMode;
  }
  panelY += 50;

  GuiLabel(Rectangle{panelX + 10, panelY, 160, 40}, "Epoch:");
  if (GuiValueBox(Rectangle{panelX + 180, panelY, 100, 40}, nullptr, &epoch, 1,
                  10000, epochEditMode)) {
    epochEditMode = !epochEditMode;
  }
  panelY += 50;

  if (!X.dim || training)
    GuiDisable();
  if (GuiButton(Rectangle{panelX + 10, panelY, (panelWidth - 30) / 2, 40},
                "Run one epoch")) {
    if (trainThread.joinable())
      trainThread.join();
    trainThread =
        std::thread(RunOneEpoch, lr, lambda, std::ref(net), std::ref(loss),
                    std::ref(splinesData), std::ref(splinesAlpha),
                    std::ref(min_act), std::ref(max_act),
                    std::ref(splinesDataMutex), std::ref(X), std::ref(y));
  }
  GuiEnable();

  if (!X.dim)
    GuiDisable();
  if (GuiButton(Rectangle{panelX + panelWidth / 2 + 5, panelY,
                          (panelWidth - 30) / 2, 40},
                training ? "Pause" : "Train")) {
    if (training) {
      training = false;
      if (trainThread.joinable())
        trainThread.join();
    } else {
      if (trainThread.joinable())
        trainThread.join();
      trainThread =
          std::thread(RunTraining, lr, lambda, epoch, std::ref(training),
                      std::ref(currentEpoch), std::ref(net), std::ref(loss),
                      std::ref(splinesData), std::ref(splinesAlpha),
                      std::ref(min_act), std::ref(max_act),
                      std::ref(splinesDataMutex), std::ref(X), std::ref(y));
    }
  }
  panelY += 50;
  GuiEnable();

  std::string infoStr = "Epoch: " + std::to_string(currentEpoch) + " - Loss: ";
  if (loss < 0)
    infoStr += "<Not training>";
  else
    infoStr += std::format("{:.4f}", loss);

  GuiLabel(Rectangle{panelX + 10, panelY, panelX - 20, 40}, infoStr.c_str());
  panelY += 50;
  float progress = currentEpoch / (float)epoch;
  GuiProgressBar(Rectangle{panelX + 10, panelY, panelWidth - 20, 30}, nullptr,
                 nullptr, &progress, 0, 1);
  panelY += 40;

  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Save Checkpoint")) {
    HandleSaveCheckpoint(net);
  }
  panelY += 50;

  if (GuiButton(Rectangle{panelX + 10, panelY, panelWidth - 20, 40},
                "Back to Menu")) {
    training = false;
    if (trainThread.joinable())
      trainThread.join();
    kanGuiState = MENU;
  }
  panelY += 50;
}

#endif
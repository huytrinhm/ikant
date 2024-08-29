#include "gui/gui.h"

KAN::KANNet net;
KAN::Tensor X, y;

float loss = -1;

std::thread trainThread;
std::atomic<bool> training(false);
std::atomic<int> currentEpoch(0);

std::vector<std::vector<std::vector<std::vector<float>>>> splinesData;
std::vector<std::vector<std::vector<float>>> splinesAlpha;
std::vector<std::vector<float>> min_act;
std::vector<std::vector<float>> max_act;
std::mutex splinesDataMutex;

int main() {
  uint32_t num_layers, spline_order, grid_size, *widths;
  float* params_data;
  KAN::KANNet_load_checkpoint("checkpoint2.dat", num_layers, spline_order,
                              grid_size, widths, params_data);
  net = KAN::KANNet_create(std::vector(widths, widths + num_layers + 1),
                           spline_order, grid_size, params_data);
  InitKANNet(net, splinesData, splinesAlpha, min_act, max_act, currentEpoch,
             loss);

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
    DrawKANNet(layers, splinesAlpha, camera, mousePosition);
    if (kanGuiState != EDIT)
      DrawKANSplines(layers, net, splinesData, splinesAlpha, min_act, max_act,
                     splinesDataMutex);
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
        DrawMenuGUI(panelWidth, panelX, panelY, kanGuiState, net, splinesData,
                    splinesAlpha, min_act, max_act, currentEpoch, loss);
        break;
      case EDIT:
        DrawEditGUI(panelWidth, panelX, panelY, kanGuiState, net, splinesData,
                    splinesAlpha, min_act, max_act, currentEpoch, loss);
        break;
      case TRAIN:
        DrawTrainGUI(panelWidth, panelX, panelY, kanGuiState, net, X, y,
                     training, loss, currentEpoch, trainThread, splinesData,
                     splinesAlpha, min_act, max_act, splinesDataMutex);
        break;
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

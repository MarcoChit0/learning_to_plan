; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 167367

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child370 - child
    bread485 bread253 - bread-portion
    content62 content317 - content-portion
    tray244 tray88 tray296 tray321 - tray
    table399 table447 table96 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at tray88 kitchen)
     (at tray296 kitchen)
     (at tray321 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_bread bread253)
     (at_kitchen_content content62)
     (at_kitchen_content content317)
     (no_gluten_bread bread253)
     (no_gluten_content content62)
     (allergic_gluten child370)
     (not_allergic_gluten child481)
     (waiting child481 table399)
     (waiting child370 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child481)
     (served child370)
    )
  )
)

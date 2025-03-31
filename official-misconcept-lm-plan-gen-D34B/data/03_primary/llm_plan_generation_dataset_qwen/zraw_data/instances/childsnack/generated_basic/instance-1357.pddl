; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 70335

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child145 child438 - child
    bread196 bread343 - bread-portion
    content16 content368 - content-portion
    tray470 tray320 tray306 - tray
    table450 table452 table489 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray470 kitchen)
     (at tray320 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread196)
     (at_kitchen_bread bread343)
     (at_kitchen_content content16)
     (at_kitchen_content content368)
     (not_allergic_gluten child438)
     (not_allergic_gluten child145)
     (waiting child145 table450)
     (waiting child438 table450)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child145)
     (served child438)
    )
  )
)

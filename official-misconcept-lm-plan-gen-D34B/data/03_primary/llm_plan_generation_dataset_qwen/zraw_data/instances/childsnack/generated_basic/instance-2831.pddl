; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 652108

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child483 child137 - child
    bread57 bread248 - bread-portion
    content228 content393 - content-portion
    tray426 tray345 - tray
    table86 table90 table450 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray345 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_bread bread248)
     (at_kitchen_content content228)
     (at_kitchen_content content393)
     (not_allergic_gluten child137)
     (not_allergic_gluten child483)
     (waiting child483 table90)
     (waiting child137 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child483)
     (served child137)
    )
  )
)

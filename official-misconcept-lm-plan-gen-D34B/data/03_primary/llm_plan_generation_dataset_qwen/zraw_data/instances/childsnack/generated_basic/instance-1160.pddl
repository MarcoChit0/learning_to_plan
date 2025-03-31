; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 138467

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 child262 - child
    bread74 bread288 - bread-portion
    content396 content375 - content-portion
    tray161 tray107 tray9 - tray
    table371 table495 table190 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray161 kitchen)
     (at tray107 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread74)
     (at_kitchen_bread bread288)
     (at_kitchen_content content396)
     (at_kitchen_content content375)
     (no_gluten_bread bread74)
     (no_gluten_content content375)
     (allergic_gluten child134)
     (not_allergic_gluten child262)
     (waiting child134 table495)
     (waiting child262 table190)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child134)
     (served child262)
    )
  )
)

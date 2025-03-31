; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 400988

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child380 child488 - child
    bread94 bread267 - bread-portion
    content103 content69 - content-portion
    tray264 tray183 tray32 tray91 - tray
    table319 table370 table295 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray264 kitchen)
     (at tray183 kitchen)
     (at tray32 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread267)
     (at_kitchen_content content103)
     (at_kitchen_content content69)
     (not_allergic_gluten child488)
     (not_allergic_gluten child380)
     (waiting child380 table370)
     (waiting child488 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child380)
     (served child488)
    )
  )
)

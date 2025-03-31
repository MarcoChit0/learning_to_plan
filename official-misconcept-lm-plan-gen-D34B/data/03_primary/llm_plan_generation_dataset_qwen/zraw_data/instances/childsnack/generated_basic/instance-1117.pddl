; child-snack task with 3 children and 0.38 gluten factor 
; constant factor of 1.5
; random seed: 584328

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child415 child432 child161 - child
    bread234 bread308 bread291 - bread-portion
    content379 content79 content276 - content-portion
    tray423 tray122 tray341 - tray
    table119 table58 table376 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray423 kitchen)
     (at tray122 kitchen)
     (at tray341 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread291)
     (at_kitchen_content content379)
     (at_kitchen_content content79)
     (at_kitchen_content content276)
     (no_gluten_bread bread234)
     (no_gluten_content content79)
     (allergic_gluten child161)
     (not_allergic_gluten child432)
     (not_allergic_gluten child415)
     (waiting child415 table58)
     (waiting child432 table119)
     (waiting child161 table376)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child415)
     (served child432)
     (served child161)
    )
  )
)

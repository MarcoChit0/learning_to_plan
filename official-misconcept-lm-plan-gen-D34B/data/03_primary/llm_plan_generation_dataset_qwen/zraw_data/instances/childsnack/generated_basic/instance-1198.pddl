; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 23114

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child369 child60 child415 - child
    bread337 bread347 bread318 - bread-portion
    content93 content13 content131 - content-portion
    tray242 tray312 tray245 - tray
    table174 table88 table156 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray242 kitchen)
     (at tray312 kitchen)
     (at tray245 kitchen)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread347)
     (at_kitchen_bread bread318)
     (at_kitchen_content content93)
     (at_kitchen_content content13)
     (at_kitchen_content content131)
     (not_allergic_gluten child369)
     (not_allergic_gluten child60)
     (not_allergic_gluten child415)
     (waiting child369 table88)
     (waiting child60 table174)
     (waiting child415 table156)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child369)
     (served child60)
     (served child415)
    )
  )
)

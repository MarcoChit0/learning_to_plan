; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 486327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child221 child63 - child
    bread168 bread52 - bread-portion
    content249 content70 - content-portion
    tray349 tray132 tray56 - tray
    table131 table499 table382 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray132 kitchen)
     (at tray56 kitchen)
     (at_kitchen_bread bread168)
     (at_kitchen_bread bread52)
     (at_kitchen_content content249)
     (at_kitchen_content content70)
     (not_allergic_gluten child221)
     (not_allergic_gluten child63)
     (waiting child221 table131)
     (waiting child63 table131)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child221)
     (served child63)
    )
  )
)

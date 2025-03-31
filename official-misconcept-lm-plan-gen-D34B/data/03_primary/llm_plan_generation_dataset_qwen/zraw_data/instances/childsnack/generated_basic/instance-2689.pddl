; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 4603

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 child383 child484 - child
    bread143 bread125 bread282 - bread-portion
    content27 content476 content168 - content-portion
    tray164 tray194 tray292 - tray
    table21 table284 table346 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at tray194 kitchen)
     (at tray292 kitchen)
     (at_kitchen_bread bread143)
     (at_kitchen_bread bread125)
     (at_kitchen_bread bread282)
     (at_kitchen_content content27)
     (at_kitchen_content content476)
     (at_kitchen_content content168)
     (not_allergic_gluten child383)
     (not_allergic_gluten child202)
     (not_allergic_gluten child484)
     (waiting child202 table346)
     (waiting child383 table346)
     (waiting child484 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child202)
     (served child383)
     (served child484)
    )
  )
)

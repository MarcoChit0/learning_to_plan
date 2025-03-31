; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 665945

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child461 child138 - child
    bread171 bread297 - bread-portion
    content65 content472 - content-portion
    tray385 tray57 tray335 tray426 - tray
    table256 table403 table276 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray385 kitchen)
     (at tray57 kitchen)
     (at tray335 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread171)
     (at_kitchen_bread bread297)
     (at_kitchen_content content65)
     (at_kitchen_content content472)
     (not_allergic_gluten child138)
     (not_allergic_gluten child461)
     (waiting child461 table256)
     (waiting child138 table276)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child461)
     (served child138)
    )
  )
)

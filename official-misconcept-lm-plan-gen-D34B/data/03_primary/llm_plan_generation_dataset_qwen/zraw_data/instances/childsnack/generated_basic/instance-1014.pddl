; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 201020

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child102 child163 - child
    bread248 bread416 bread433 - bread-portion
    content125 content18 content300 - content-portion
    tray452 - tray
    table227 table176 table102 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray452 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_bread bread416)
     (at_kitchen_bread bread433)
     (at_kitchen_content content125)
     (at_kitchen_content content18)
     (at_kitchen_content content300)
     (not_allergic_gluten child151)
     (not_allergic_gluten child102)
     (not_allergic_gluten child163)
     (waiting child151 table227)
     (waiting child102 table102)
     (waiting child163 table176)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child151)
     (served child102)
     (served child163)
    )
  )
)

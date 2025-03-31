; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 406768

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 child71 child474 - child
    bread97 bread39 bread207 - bread-portion
    content413 content133 content28 - content-portion
    tray142 tray67 tray306 - tray
    table416 table441 table435 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at tray67 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_bread bread39)
     (at_kitchen_bread bread207)
     (at_kitchen_content content413)
     (at_kitchen_content content133)
     (at_kitchen_content content28)
     (not_allergic_gluten child439)
     (not_allergic_gluten child474)
     (not_allergic_gluten child71)
     (waiting child439 table416)
     (waiting child71 table441)
     (waiting child474 table441)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child439)
     (served child71)
     (served child474)
    )
  )
)

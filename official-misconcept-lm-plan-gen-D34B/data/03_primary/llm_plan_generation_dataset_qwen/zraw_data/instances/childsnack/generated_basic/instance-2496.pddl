; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 241520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child87 child478 - child
    bread496 bread142 - bread-portion
    content106 content131 - content-portion
    tray89 tray1 - tray
    table260 table115 table424 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray89 kitchen)
     (at tray1 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_bread bread142)
     (at_kitchen_content content106)
     (at_kitchen_content content131)
     (not_allergic_gluten child87)
     (not_allergic_gluten child478)
     (waiting child87 table424)
     (waiting child478 table424)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child87)
     (served child478)
    )
  )
)

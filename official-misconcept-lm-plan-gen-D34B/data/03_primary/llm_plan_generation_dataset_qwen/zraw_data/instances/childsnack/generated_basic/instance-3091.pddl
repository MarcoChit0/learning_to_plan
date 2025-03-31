; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 838251

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 child159 - child
    bread435 bread89 - bread-portion
    content283 content83 - content-portion
    tray8 tray263 tray184 - tray
    table460 table197 table280 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray263 kitchen)
     (at tray184 kitchen)
     (at_kitchen_bread bread435)
     (at_kitchen_bread bread89)
     (at_kitchen_content content283)
     (at_kitchen_content content83)
     (not_allergic_gluten child114)
     (not_allergic_gluten child159)
     (waiting child114 table280)
     (waiting child159 table460)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child114)
     (served child159)
    )
  )
)

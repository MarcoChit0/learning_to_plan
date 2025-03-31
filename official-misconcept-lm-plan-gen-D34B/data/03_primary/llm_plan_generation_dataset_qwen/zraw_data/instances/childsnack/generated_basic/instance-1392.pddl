; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 959795

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child471 child238 - child
    bread462 bread446 - bread-portion
    content251 content190 - content-portion
    tray380 - tray
    table331 table7 table73 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at_kitchen_bread bread462)
     (at_kitchen_bread bread446)
     (at_kitchen_content content251)
     (at_kitchen_content content190)
     (no_gluten_bread bread462)
     (no_gluten_content content190)
     (allergic_gluten child238)
     (not_allergic_gluten child471)
     (waiting child471 table73)
     (waiting child238 table73)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child471)
     (served child238)
    )
  )
)

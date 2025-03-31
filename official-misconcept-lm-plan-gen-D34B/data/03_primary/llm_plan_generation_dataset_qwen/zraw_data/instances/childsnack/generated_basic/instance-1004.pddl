; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 348579

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 child453 child111 - child
    bread135 bread133 bread473 - bread-portion
    content391 content484 content84 - content-portion
    tray274 - tray
    table261 table491 table240 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray274 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread133)
     (at_kitchen_bread bread473)
     (at_kitchen_content content391)
     (at_kitchen_content content484)
     (at_kitchen_content content84)
     (not_allergic_gluten child155)
     (not_allergic_gluten child453)
     (not_allergic_gluten child111)
     (waiting child155 table491)
     (waiting child453 table240)
     (waiting child111 table491)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child155)
     (served child453)
     (served child111)
    )
  )
)

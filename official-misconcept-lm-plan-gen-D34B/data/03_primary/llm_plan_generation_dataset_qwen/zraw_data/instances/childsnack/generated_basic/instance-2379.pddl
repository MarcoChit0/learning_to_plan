; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 380761

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child285 - child
    bread342 bread404 - bread-portion
    content87 content480 - content-portion
    tray279 - tray
    table147 table246 table208 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_bread bread404)
     (at_kitchen_content content87)
     (at_kitchen_content content480)
     (not_allergic_gluten child495)
     (not_allergic_gluten child285)
     (waiting child495 table208)
     (waiting child285 table208)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child285)
    )
  )
)

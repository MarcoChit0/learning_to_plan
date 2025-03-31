; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 39628

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 child200 - child
    bread400 bread435 - bread-portion
    content165 content411 - content-portion
    tray188 tray152 tray123 - tray
    table337 table77 table190 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray152 kitchen)
     (at tray123 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_bread bread435)
     (at_kitchen_content content165)
     (at_kitchen_content content411)
     (not_allergic_gluten child200)
     (not_allergic_gluten child354)
     (waiting child354 table337)
     (waiting child200 table77)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child354)
     (served child200)
    )
  )
)

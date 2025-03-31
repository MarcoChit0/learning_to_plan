; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 78133

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 - child
    bread32 - bread-portion
    content427 - content-portion
    tray244 - tray
    table219 table226 table192 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_content content427)
     (not_allergic_gluten child460)
     (waiting child460 table226)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child460)
    )
  )
)

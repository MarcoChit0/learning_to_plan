; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 345396

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 child141 - child
    bread386 bread279 - bread-portion
    content386 content161 - content-portion
    tray48 tray3 tray43 tray319 - tray
    table366 table417 table238 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray3 kitchen)
     (at tray43 kitchen)
     (at tray319 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_bread bread279)
     (at_kitchen_content content386)
     (at_kitchen_content content161)
     (not_allergic_gluten child460)
     (not_allergic_gluten child141)
     (waiting child460 table417)
     (waiting child141 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child460)
     (served child141)
    )
  )
)

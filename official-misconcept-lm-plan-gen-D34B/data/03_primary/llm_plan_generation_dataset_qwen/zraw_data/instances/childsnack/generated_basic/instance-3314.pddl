; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 632666

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child178 child338 - child
    bread250 bread132 - bread-portion
    content206 content51 - content-portion
    tray55 tray27 - tray
    table190 table492 table310 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at tray27 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread132)
     (at_kitchen_content content206)
     (at_kitchen_content content51)
     (not_allergic_gluten child338)
     (not_allergic_gluten child178)
     (waiting child178 table190)
     (waiting child338 table310)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child178)
     (served child338)
    )
  )
)

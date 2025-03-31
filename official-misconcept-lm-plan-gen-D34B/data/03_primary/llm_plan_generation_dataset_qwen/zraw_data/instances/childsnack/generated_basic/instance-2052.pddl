; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 451344

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child299 child206 - child
    bread434 bread340 - bread-portion
    content413 content14 - content-portion
    tray243 tray474 tray75 - tray
    table134 table262 table337 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at tray474 kitchen)
     (at tray75 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_bread bread340)
     (at_kitchen_content content413)
     (at_kitchen_content content14)
     (not_allergic_gluten child206)
     (not_allergic_gluten child299)
     (waiting child299 table262)
     (waiting child206 table262)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child299)
     (served child206)
    )
  )
)

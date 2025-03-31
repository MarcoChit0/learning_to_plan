; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 177575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child262 - child
    bread343 bread163 - bread-portion
    content419 content460 - content-portion
    tray446 tray223 - tray
    table387 table139 table370 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread163)
     (at_kitchen_content content419)
     (at_kitchen_content content460)
     (not_allergic_gluten child120)
     (not_allergic_gluten child262)
     (waiting child120 table387)
     (waiting child262 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child120)
     (served child262)
    )
  )
)

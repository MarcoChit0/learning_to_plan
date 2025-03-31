; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 22901

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child450 - child
    bread359 bread300 - bread-portion
    content41 content21 - content-portion
    tray83 tray80 - tray
    table57 table327 table373 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray80 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_bread bread300)
     (at_kitchen_content content41)
     (at_kitchen_content content21)
     (not_allergic_gluten child450)
     (not_allergic_gluten child495)
     (waiting child495 table373)
     (waiting child450 table327)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child450)
    )
  )
)

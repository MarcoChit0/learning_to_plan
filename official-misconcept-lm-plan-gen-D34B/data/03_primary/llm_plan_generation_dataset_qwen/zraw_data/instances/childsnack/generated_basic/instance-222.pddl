; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 837310

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child435 child152 - child
    bread105 bread238 - bread-portion
    content390 content240 - content-portion
    tray305 tray33 tray151 - tray
    table113 table300 table61 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray33 kitchen)
     (at tray151 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_bread bread238)
     (at_kitchen_content content390)
     (at_kitchen_content content240)
     (not_allergic_gluten child435)
     (not_allergic_gluten child152)
     (waiting child435 table300)
     (waiting child152 table61)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child435)
     (served child152)
    )
  )
)

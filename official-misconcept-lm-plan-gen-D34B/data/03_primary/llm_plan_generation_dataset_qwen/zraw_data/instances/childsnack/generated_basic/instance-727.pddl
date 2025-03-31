; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 35222

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 - child
    bread414 - bread-portion
    content373 - content-portion
    tray321 tray196 tray390 tray222 - tray
    table30 table14 table73 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at tray196 kitchen)
     (at tray390 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread414)
     (at_kitchen_content content373)
     (not_allergic_gluten child231)
     (waiting child231 table30)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child231)
    )
  )
)

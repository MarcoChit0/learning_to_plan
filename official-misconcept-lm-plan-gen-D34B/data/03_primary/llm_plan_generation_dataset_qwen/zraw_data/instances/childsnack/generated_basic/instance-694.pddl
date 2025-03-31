; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 405706

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 - child
    bread498 - bread-portion
    content33 - content-portion
    tray62 tray295 tray426 tray454 - tray
    table61 table183 table294 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray62 kitchen)
     (at tray295 kitchen)
     (at tray426 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content33)
     (not_allergic_gluten child200)
     (waiting child200 table294)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child200)
    )
  )
)

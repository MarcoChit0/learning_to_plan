; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 11997

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 - child
    bread50 - bread-portion
    content224 - content-portion
    tray75 tray409 tray306 - tray
    table448 table307 table252 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray75 kitchen)
     (at tray409 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_content content224)
     (not_allergic_gluten child68)
     (waiting child68 table448)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child68)
    )
  )
)

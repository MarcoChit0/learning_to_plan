; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 857707

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 - child
    bread78 - bread-portion
    content92 - content-portion
    tray215 - tray
    table381 table191 table261 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_content content92)
     (not_allergic_gluten child289)
     (waiting child289 table191)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child289)
    )
  )
)

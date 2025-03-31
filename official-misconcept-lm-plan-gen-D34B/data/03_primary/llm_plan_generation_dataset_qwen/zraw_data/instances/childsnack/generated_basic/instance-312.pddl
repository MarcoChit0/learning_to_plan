; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 838676

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 - child
    bread403 - bread-portion
    content314 - content-portion
    tray345 - tray
    table246 table315 table271 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at_kitchen_bread bread403)
     (at_kitchen_content content314)
     (not_allergic_gluten child229)
     (waiting child229 table271)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child229)
    )
  )
)

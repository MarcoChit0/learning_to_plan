; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 787785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 - child
    bread252 - bread-portion
    content106 - content-portion
    tray145 tray53 tray369 - tray
    table422 table141 table264 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray145 kitchen)
     (at tray53 kitchen)
     (at tray369 kitchen)
     (at_kitchen_bread bread252)
     (at_kitchen_content content106)
     (not_allergic_gluten child227)
     (waiting child227 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child227)
    )
  )
)

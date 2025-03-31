; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 836363

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 - child
    bread410 - bread-portion
    content119 - content-portion
    tray379 tray198 tray118 tray80 - tray
    table177 table273 table197 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray379 kitchen)
     (at tray198 kitchen)
     (at tray118 kitchen)
     (at tray80 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_content content119)
     (not_allergic_gluten child152)
     (waiting child152 table177)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child152)
    )
  )
)

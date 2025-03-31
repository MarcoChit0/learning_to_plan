; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 710142

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 - child
    bread53 - bread-portion
    content425 - content-portion
    tray69 tray136 tray83 tray325 - tray
    table286 table146 table295 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray69 kitchen)
     (at tray136 kitchen)
     (at tray83 kitchen)
     (at tray325 kitchen)
     (at_kitchen_bread bread53)
     (at_kitchen_content content425)
     (not_allergic_gluten child181)
     (waiting child181 table146)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child181)
    )
  )
)

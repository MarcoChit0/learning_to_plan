; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 432466

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child430 - child
    bread95 - bread-portion
    content137 - content-portion
    tray339 tray149 tray402 tray206 - tray
    table98 table298 table487 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray339 kitchen)
     (at tray149 kitchen)
     (at tray402 kitchen)
     (at tray206 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_content content137)
     (not_allergic_gluten child430)
     (waiting child430 table298)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child430)
    )
  )
)

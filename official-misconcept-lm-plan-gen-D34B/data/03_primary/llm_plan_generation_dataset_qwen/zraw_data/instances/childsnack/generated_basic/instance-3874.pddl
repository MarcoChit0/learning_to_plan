; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 194329

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child154 - child
    bread475 - bread-portion
    content12 - content-portion
    tray274 tray85 - tray
    table66 table444 table233 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray274 kitchen)
     (at tray85 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_content content12)
     (not_allergic_gluten child154)
     (waiting child154 table444)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child154)
    )
  )
)

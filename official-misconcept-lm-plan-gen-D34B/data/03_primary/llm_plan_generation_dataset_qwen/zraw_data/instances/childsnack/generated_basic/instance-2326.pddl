; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 879042

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child14 - child
    bread114 - bread-portion
    content187 - content-portion
    tray298 - tray
    table368 table122 table371 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_content content187)
     (not_allergic_gluten child14)
     (waiting child14 table368)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child14)
    )
  )
)

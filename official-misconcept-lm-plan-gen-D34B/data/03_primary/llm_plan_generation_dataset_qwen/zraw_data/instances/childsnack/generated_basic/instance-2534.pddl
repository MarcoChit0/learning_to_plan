; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 140698

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child470 - child
    bread315 - bread-portion
    content17 - content-portion
    tray327 tray187 tray424 - tray
    table73 table306 table176 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at tray187 kitchen)
     (at tray424 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_content content17)
     (not_allergic_gluten child470)
     (waiting child470 table176)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child470)
    )
  )
)

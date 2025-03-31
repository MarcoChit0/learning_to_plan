; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 742079

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child480 - child
    bread229 - bread-portion
    content326 - content-portion
    tray499 tray205 tray464 - tray
    table352 table207 table376 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray205 kitchen)
     (at tray464 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_content content326)
     (not_allergic_gluten child480)
     (waiting child480 table376)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child480)
    )
  )
)

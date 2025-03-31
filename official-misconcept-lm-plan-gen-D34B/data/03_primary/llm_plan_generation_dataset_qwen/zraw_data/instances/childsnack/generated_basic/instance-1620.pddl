; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 844639

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread177 - bread-portion
    content145 - content-portion
    tray403 tray288 tray433 tray160 - tray
    table84 table299 table11 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at tray288 kitchen)
     (at tray433 kitchen)
     (at tray160 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_content content145)
     (not_allergic_gluten child431)
     (waiting child431 table84)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)

; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 987227

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 child299 - child
    bread477 bread112 - bread-portion
    content121 content60 - content-portion
    tray376 tray140 tray285 - tray
    table2 table128 table84 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at tray140 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread477)
     (at_kitchen_bread bread112)
     (at_kitchen_content content121)
     (at_kitchen_content content60)
     (not_allergic_gluten child114)
     (not_allergic_gluten child299)
     (waiting child114 table84)
     (waiting child299 table84)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child114)
     (served child299)
    )
  )
)

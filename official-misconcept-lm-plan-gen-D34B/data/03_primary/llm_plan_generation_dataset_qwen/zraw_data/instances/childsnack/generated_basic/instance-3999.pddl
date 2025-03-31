; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 937549

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 - child
    bread439 - bread-portion
    content115 - content-portion
    tray408 tray158 tray87 - tray
    table417 table211 table496 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at tray158 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread439)
     (at_kitchen_content content115)
     (not_allergic_gluten child142)
     (waiting child142 table211)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child142)
    )
  )
)

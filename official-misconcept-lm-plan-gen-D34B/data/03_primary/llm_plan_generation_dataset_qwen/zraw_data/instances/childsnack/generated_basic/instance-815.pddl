; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 602872

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 - child
    bread103 - bread-portion
    content229 - content-portion
    tray211 tray123 - tray
    table38 table456 table342 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray211 kitchen)
     (at tray123 kitchen)
     (at_kitchen_bread bread103)
     (at_kitchen_content content229)
     (not_allergic_gluten child82)
     (waiting child82 table38)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child82)
    )
  )
)

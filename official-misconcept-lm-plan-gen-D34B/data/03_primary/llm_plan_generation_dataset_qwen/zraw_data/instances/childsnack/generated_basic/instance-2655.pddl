; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 461012

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 - child
    bread26 - bread-portion
    content65 - content-portion
    tray321 tray248 tray211 tray158 - tray
    table441 table117 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at tray248 kitchen)
     (at tray211 kitchen)
     (at tray158 kitchen)
     (at_kitchen_bread bread26)
     (at_kitchen_content content65)
     (not_allergic_gluten child279)
     (waiting child279 table117)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child279)
    )
  )
)

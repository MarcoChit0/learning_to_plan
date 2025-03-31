; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 386366

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 - child
    bread407 - bread-portion
    content21 - content-portion
    tray53 tray207 tray255 tray248 - tray
    table318 table252 table217 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray207 kitchen)
     (at tray255 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_content content21)
     (not_allergic_gluten child425)
     (waiting child425 table217)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child425)
    )
  )
)

; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 512919

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child19 - child
    bread210 - bread-portion
    content113 - content-portion
    tray57 - tray
    table491 table474 table285 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_content content113)
     (not_allergic_gluten child19)
     (waiting child19 table474)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child19)
    )
  )
)

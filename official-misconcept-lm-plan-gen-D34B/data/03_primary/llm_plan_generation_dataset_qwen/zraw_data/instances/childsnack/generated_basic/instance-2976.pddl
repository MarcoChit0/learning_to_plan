; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 685946

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child19 - child
    bread228 - bread-portion
    content76 - content-portion
    tray179 tray27 tray282 tray18 - tray
    table474 table390 table344 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray179 kitchen)
     (at tray27 kitchen)
     (at tray282 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread228)
     (at_kitchen_content content76)
     (not_allergic_gluten child19)
     (waiting child19 table344)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child19)
    )
  )
)

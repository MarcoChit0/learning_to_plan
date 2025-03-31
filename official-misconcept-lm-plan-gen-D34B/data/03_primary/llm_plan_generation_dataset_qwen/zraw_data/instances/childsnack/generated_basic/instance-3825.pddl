; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 995988

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child470 - child
    bread336 - bread-portion
    content263 - content-portion
    tray59 tray62 tray360 tray405 - tray
    table302 table59 table200 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at tray62 kitchen)
     (at tray360 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread336)
     (at_kitchen_content content263)
     (not_allergic_gluten child470)
     (waiting child470 table302)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child470)
    )
  )
)

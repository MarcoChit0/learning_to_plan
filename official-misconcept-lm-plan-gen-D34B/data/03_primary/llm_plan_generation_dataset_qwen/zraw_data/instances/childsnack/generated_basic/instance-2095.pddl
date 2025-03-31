; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 90251

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 - child
    bread422 - bread-portion
    content454 - content-portion
    tray283 tray307 tray244 tray467 - tray
    table367 table2 table454 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray307 kitchen)
     (at tray244 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_content content454)
     (not_allergic_gluten child168)
     (waiting child168 table454)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child168)
    )
  )
)

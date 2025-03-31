; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 666187

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child300 - child
    bread352 - bread-portion
    content248 - content-portion
    tray308 tray150 tray375 - tray
    table185 table304 table74 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at tray150 kitchen)
     (at tray375 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_content content248)
     (not_allergic_gluten child300)
     (waiting child300 table74)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child300)
    )
  )
)

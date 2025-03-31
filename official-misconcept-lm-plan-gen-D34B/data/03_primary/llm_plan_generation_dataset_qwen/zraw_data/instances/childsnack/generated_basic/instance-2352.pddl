; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 169035

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 - child
    bread170 - bread-portion
    content146 - content-portion
    tray224 - tray
    table155 table400 table113 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray224 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_content content146)
     (not_allergic_gluten child204)
     (waiting child204 table113)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child204)
    )
  )
)

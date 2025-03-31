; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 703849

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child418 - child
    bread221 - bread-portion
    content351 - content-portion
    tray215 tray267 tray140 tray214 - tray
    table125 table40 table88 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray267 kitchen)
     (at tray140 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_content content351)
     (not_allergic_gluten child418)
     (waiting child418 table40)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child418)
    )
  )
)

; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 172949

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child98 - child
    bread106 - bread-portion
    content99 - content-portion
    tray255 tray338 tray210 tray84 - tray
    table219 table454 table151 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray255 kitchen)
     (at tray338 kitchen)
     (at tray210 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_content content99)
     (not_allergic_gluten child98)
     (waiting child98 table219)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child98)
    )
  )
)

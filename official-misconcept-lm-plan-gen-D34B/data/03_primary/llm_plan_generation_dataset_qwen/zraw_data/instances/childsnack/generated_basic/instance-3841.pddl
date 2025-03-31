; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 326739

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 - child
    bread245 - bread-portion
    content282 - content-portion
    tray302 tray206 tray155 tray100 - tray
    table397 table371 table375 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray206 kitchen)
     (at tray155 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread245)
     (at_kitchen_content content282)
     (not_allergic_gluten child198)
     (waiting child198 table397)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child198)
    )
  )
)

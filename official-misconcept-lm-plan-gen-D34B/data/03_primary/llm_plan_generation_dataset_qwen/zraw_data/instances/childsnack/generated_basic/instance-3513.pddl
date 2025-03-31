; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 130171

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 child335 - child
    bread219 bread231 - bread-portion
    content313 content189 - content-portion
    tray415 tray267 - tray
    table179 table4 table41 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray415 kitchen)
     (at tray267 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_bread bread231)
     (at_kitchen_content content313)
     (at_kitchen_content content189)
     (not_allergic_gluten child335)
     (not_allergic_gluten child65)
     (waiting child65 table41)
     (waiting child335 table179)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child65)
     (served child335)
    )
  )
)
